//////
/// Commands
//////

import * as vscode from "vscode";
import * as YAML from "yaml";
import * as path from "path";

import { log } from "./logging";
import { DelphyneServer } from "./server";
import { Diagnostic, Trace, TraceAnswerId } from "./stubs/feedback";
import { PointedTree, TreeInfo, TreeView } from "./tree_view";
import { CommandElement, SerializedCommand } from "./elements";
import {
  parseYamlWithLocInfo,
  prettyYaml,
  serializeWithoutLocInfo,
} from "./yaml_utils";
import { getLocalExecutionContext, getWorkspaceRoot } from "./config";
import { ROOT_ID } from "./common";
import { getEditorForUri } from "./edit_utils";

const DEFAULT_COMMAND = { command: null, args: {} };
const EXECUTE_COMMAND_ENDPOINT = "execute-command";
const RUN_STRATEGY_DEFAULT_NUM_REQUESTS = 10;
const DELPHYNE_COMMAND_FILE_EXTENSION = ".exec.yaml";
const DELPHYNE_COMMAND_HEADER = "delphyne-command";

// TODO: this is hardcoded for now
function taskOptions(name: string, args: { [key: string]: any }) {
  return {
    sync_period_ms: 100,
    sync_by_default:
      name === "test_command" ||
      (name === "answer_query" && args["completions"] === 1),
  };
}

//////
/// Command Definition
//////

// TODO: do proper validation

export interface CommandResultFromServer {
  diagnostics: Diagnostic[];
  result: any;
}

type CommandStatus = "pending" | "completed" | "interrupted";

export interface CommandOutcome {
  status: CommandStatus;
  diagnostics: Diagnostic[];
  result: any;
}

export interface CommandSpec {
  command: string;
  args: { [key: string]: any };
}

export interface CommandFile extends CommandSpec {
  outcome?: CommandOutcome;
}

//////
/// Commands Manager
//////

export class CommandsManager implements vscode.CodeActionProvider {
  constructor(
    private treeView: TreeView,
    private server: DelphyneServer,
  ) {
    this.register();
  }
  private running = new Set<vscode.Uri>();

  private register() {
    vscode.commands.registerCommand(
      "delphyne.createCommandBuffer",
      (cmd: CommandSpec | null, execute) =>
        createCommandBuffer(cmd ?? DEFAULT_COMMAND, execute),
    );
    vscode.commands.registerCommand("delphyne.runStrategy", runStrategy);
    vscode.commands.registerCommand(
      "delphyne.executeCommand",
      this.executeCommand,
      this,
    );
    vscode.commands.registerCommand(
      "delphyne.clearOutput",
      this.clearOutput,
      this,
    );
    vscode.commands.registerCommand("delphyne.showTrace", this.showTrace, this);
    vscode.languages.registerCodeActionsProvider("yaml", this);
  }

  public provideCodeActions(
    document: vscode.TextDocument,
    range: vscode.Range,
  ): vscode.CodeAction[] | undefined {
    if (!isCommandFile(document)) {
      return [];
    }
    // If the command is not launched, we can execute it.
    // We cannot execute it if it is running already.
    // We can also clone the command.
    const executeCommand = new vscode.CodeAction(
      "Execute Command",
      vscode.CodeActionKind.Empty,
    );
    executeCommand.command = {
      command: "delphyne.executeCommand",
      title: "Execute Command",
      arguments: [document],
    };
    const clearOutput = new vscode.CodeAction(
      "Clear Output",
      vscode.CodeActionKind.Empty,
    );
    clearOutput.command = {
      command: "delphyne.clearOutput",
      title: "Clear Output",
      arguments: [document],
    };
    const all: (vscode.CodeAction | null)[] = [
      this.showTraceAction(document),
      executeCommand,
      clearOutput,
    ];
    return all.filter((a): a is vscode.CodeAction => a !== null);
  }

  async executeCommand(document: vscode.TextDocument) {
    const uri = document.uri;
    if (this.running.has(uri)) {
      vscode.window.showInformationMessage(
        "The command in this file is already running. You can cancel it and try again.",
      );
      log.info(`Command ${uri} already running`);
      return;
    }
    this.running.add(uri);
    const context = getLocalExecutionContext(document);
    const parsed = YAML.parse(document.getText()) as CommandFile;
    const name = parsed.command;
    const arg = uriBase(uri);
    const origin: CommandElement = {
      kind: "command",
      uri: uri,
      command: serializeCommand(parsed),
    };
    const query = EXECUTE_COMMAND_ENDPOINT;
    const payload = {
      spec: parsed,
      context: context,
      workspace_root: getWorkspaceRoot(document.fileName),
    };

    const updateEditor = async (
      result: CommandResultFromServer | null,
      status: CommandStatus,
    ) => {
      let outcome: CommandOutcome;
      if (result === null) {
        outcome = {
          status,
          diagnostics: [],
          result: null,
        };
      } else {
        outcome = {
          status,
          diagnostics: result.diagnostics,
          result: result.result,
        };
      }

      // This is needed in case the document is closed and reopened, in which case the old
      // document object becomes invalid.
      const document = await vscode.workspace.openTextDocument(uri);
      const outcomeObj = { outcome: outcome };
      const outcomeYaml = "\n" + prettyYaml(outcomeObj);
      await clearAfterOutcomeAndAppend(document, outcomeYaml);
    };

    await updateEditor(null, "pending");
    const task = this.server.launchTask(
      { name, arg, origin },
      {
        ...taskOptions(parsed.command, parsed.args),
        handle_result: async (r: CommandResultFromServer) =>
          updateEditor(r, "pending"),
      },
      query,
      payload,
    );
    const outcome = await task.outcome;
    const status = outcome.cancelled ? "interrupted" : "completed";
    await updateEditor(outcome.result, status);
    this.running.delete(uri);
  }

  async clearOutput(document: vscode.TextDocument) {
    await clearAfterOutcomeAndAppend(document, "");
  }

  showTraceAction(document: vscode.TextDocument): vscode.CodeAction | null {
    try {
      const parsed = YAML.parse(document.getText()) as any;
      const trace = parsed.outcome.result.browsable_trace as Trace;
      if (!trace) {
        return null;
      }
      const serialized = serializeWithoutLocInfo({
        command: parsed.command,
        args: parsed.args,
      });
      const action = new vscode.CodeAction(
        "Show Trace",
        vscode.CodeActionKind.Empty,
      );
      action.command = {
        command: "delphyne.showTrace",
        title: "Show Trace",
        arguments: [document, trace, serialized],
      };
      return action;
    } catch (e) {
      return null;
    }
  }

  showTrace(document: vscode.TextDocument, trace: Trace, serialized: string) {
    const origin: CommandElement = {
      kind: "command",
      uri: document.uri,
      command: serialized,
    };
    const treeInfo = new TreeInfo(trace, origin);
    this.treeView.setPointedTree(new PointedTree(treeInfo, ROOT_ID));
  }
}

function uriBase(uri: vscode.Uri): string {
  const fileName = path.basename(uri.fsPath);
  if (uri.scheme === "untitled") {
    return fileName;
  }
  const baseName = fileName.split(".")[0];
  return baseName;
}

async function clearAfterOutcomeAndAppend(
  document: vscode.TextDocument,
  textToAppend: string,
) {
  const lines = document.getText().split("\n");
  const outcomeIdx = lines.findIndex((line) => line.trim() === "outcome:");

  // If "outcome:" is found, keep everything before it, otherwise keep the whole document
  const baseContent =
    outcomeIdx === -1
      ? lines.join("\n")
      : lines.slice(0, outcomeIdx).join("\n");

  const newContent = baseContent + textToAppend;

  const edit = new vscode.WorkspaceEdit();
  const fullRange = new vscode.Range(
    document.positionAt(0),
    document.positionAt(document.getText().length),
  );
  edit.replace(document.uri, fullRange, newContent);
  await vscode.workspace.applyEdit(edit);
}

function serializeCommand(cmd: CommandSpec): SerializedCommand {
  return serializeWithoutLocInfo(cmd);
}

async function createCommandBuffer(cmd: unknown, execute: boolean = false) {
  // Creates a new tab with the template
  const document = await vscode.workspace.openTextDocument({
    content: "# " + DELPHYNE_COMMAND_HEADER + "\n\n" + prettyYaml(cmd),
    language: "yaml",
  });
  await vscode.window.showTextDocument(document, {
    viewColumn: vscode.ViewColumn.Beside,
  });
  if (execute) {
    await vscode.commands.executeCommand("delphyne.executeCommand", document);
  }
}

function isCommandFile(document: vscode.TextDocument) {
  // Return `true` if the file starts with a number of empty lines and comments
  // (starting with `#`), one of them being `# ${DELPHYNE_COMMAND_HEADER}`.
  // Also return true if the extension is `*.exec.yaml`.
  if (document.uri.path.endsWith(DELPHYNE_COMMAND_FILE_EXTENSION)) {
    return true;
  }
  const lines = document.getText().split("\n");
  let foundHeader = false;
  for (const line of lines) {
    if (line.trim() === "") {
      continue;
    }
    if (line.trim().startsWith("#")) {
      if (line.trim().startsWith("# " + DELPHYNE_COMMAND_HEADER)) {
        return true;
      }
    } else {
      break;
    }
  }
  return false;
}

async function runStrategy() {
  const cmd = {
    command: "run_strategy",
    args: {
      strategy: "<strategy_name>",
      args: {},
      policy: "<policy_name>",
      policy_args: {},
      num_generated: 1,
      budget: { num_requests: RUN_STRATEGY_DEFAULT_NUM_REQUESTS },
    },
  };
  await createCommandBuffer(cmd);
}

export function gotoCommandResultAnswer(
  uri: vscode.Uri,
  answerIndex: TraceAnswerId,
) {
  const editor = getEditorForUri(uri);
  if (!editor) {
    return;
  }
  const parsed = parseYamlWithLocInfo(editor.document.getText()) as any;
  try {
    const queries = parsed.outcome.result.raw_trace.queries;
    for (const q of queries) {
      const key = String(answerIndex);
      if (q.answers.hasOwnProperty(key)) {
        const range = q.answers[`__loc__${key}`];
        if (range) {
          editor.selection = new vscode.Selection(range.start, range.end);
          editor.revealRange(range);
          return;
        }
      }
    }
  } catch (e) {}
}

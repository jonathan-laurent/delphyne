//////
/// Tree View Commands
//////

import * as vscode from "vscode";
import {
  NodeViewItem,
  QueryKey,
  TreeView,
  queryDemoKey,
  PropertyItem,
  queryKey,
  AnswerItem,
  copiableValue,
} from "./tree_view";
import { DemosManager } from "./demos_manager";
import {
  Query,
  StrategyDemoFeedback,
  TraceAnswerId,
  TraceNodeId,
} from "./stubs/feedback";
import { log } from "./logging";
import {
  alertIfEditorNotTwoSpaceIndent,
  getEditorForUri,
  replaceYamlValue,
} from "./edit_utils";
import { StrategyDemo } from "./stubs/demos";
import { Element } from "./elements";
import { gotoCommandResultAnswer } from "./commands";
import { addQueries } from "./edit";
import { prettyYaml } from "./yaml_utils";

export function registerTreeViewCommands(
  context: vscode.ExtensionContext,
  treeView: TreeView,
  demosManager: DemosManager,
) {
  context.subscriptions.push(
    vscode.commands.registerCommand(
      "delphyne.addQueryFromTreeView",
      (arg: NodeViewItem) => {
        const prop_item = arg as PropertyItem;
        const query = prop_item.prop as Query;
        addQueryFromTreeView(query, demosManager, treeView);
      },
    ),
    vscode.commands.registerCommand(
      "delphyne.updateQueryArgsFromTreeView",
      (arg: NodeViewItem) => {
        const prop_item = arg as PropertyItem;
        const query = prop_item.prop as Query;
        updateQueryArgsFromTreeView(query, demosManager, treeView);
      },
    ),
    vscode.commands.registerCommand(
      "delphyne.gotoQuery",
      (arg: NodeViewItem) => {
        const prop_item = arg as PropertyItem;
        const query = prop_item.prop as Query;
        gotoQuery(query, demosManager, treeView);
      },
    ),
    vscode.commands.registerCommand(
      "delphyne.gotoAnswer",
      (arg: NodeViewItem) => {
        const answer_item = arg as AnswerItem;
        gotoAnswer(answer_item.answer_id, demosManager, treeView);
      },
    ),
    vscode.commands.registerCommand(
      "delphyne.viewNode",
      (arg: { node_id: TraceNodeId }) => {
        viewNode(arg.node_id, treeView);
      },
    ),
    vscode.commands.registerCommand(
      "delphyne.copyArgValue",
      async (arg: any) => {
        const toCopy = copiableValue(arg);
        await vscode.env.clipboard.writeText(toCopy);
      },
    ),
  );
}

function getDemoInfoForTreeView(
  demosManager: DemosManager,
  treeView: TreeView,
): [StrategyDemo, StrategyDemoFeedback, vscode.TextEditor] | undefined {
  const origin = treeView.getPointedTree()?.tree.origin;
  if (!origin || origin.kind !== "strategy_demo") {
    log.error(
      "delphyne.addQueryFromTreeView: the tree view is not attached to a demo.",
    );
    return;
  }
  const demonstration = demosManager.getDemonstration(
    origin.uri,
    origin.demo,
  ) as StrategyDemo;
  if (!demonstration) {
    log.error("delphyne.addQueryFromTreeView: failed to obtain demonstration.");
    return;
  }
  const feedback = demosManager.getFeedback(
    origin.uri,
    origin.demo,
  ) as StrategyDemoFeedback;
  if (!feedback) {
    log.error("delphyne.addQueryFromTreeView: failed to obtain feedback.");
    return;
  }
  const editor = getEditorForUri(origin.uri);
  if (!editor) {
    log.warn("No editor associated with", origin.uri);
    return;
  }
  return [demonstration, feedback, editor];
}

function addQueryFromTreeView(
  query: Query,
  demosManager: DemosManager,
  treeView: TreeView,
) {
  const demoInfo = getDemoInfoForTreeView(demosManager, treeView);
  if (!demoInfo) {
    return;
  }
  const [demonstration, _, editor] = demoInfo;
  addQueries(
    [{ name: query.name, args: query.args, answer: null }],
    demonstration,
    editor,
  );
}

function queryIndex(key: QueryKey, demo: StrategyDemo): number {
  return demo.queries.findIndex((q) => queryDemoKey(q) === key);
}

function gotoQuery(
  query: Query,
  demosManager: DemosManager,
  treeView: TreeView,
) {
  const demoInfo = getDemoInfoForTreeView(demosManager, treeView);
  if (!demoInfo) {
    return;
  }
  const [demonstration, _, editor] = demoInfo;
  const idx = queryIndex(queryKey(query), demonstration);
  if (idx < 0) {
    log.error("Failed to find the query index to jump to.");
    return;
  }
  const range = demonstration.__loc_items__queries[idx];
  editor.selection = new vscode.Selection(range.start, range.end);
  editor.revealRange(range);
}

function gotoAnswer(
  answer: TraceAnswerId,
  demosManager: DemosManager,
  treeView: TreeView,
) {
  // There are different strategies for jumping to an answer depending on
  // whether the trace originates from a command or a demo. In the case it comes
  // from a demo, we use the `answer_refs` field from the server feedback to
  // convert the trace answer id to a query demo index and answer index.
  if (treeView.getPointedTree()?.tree.origin?.kind === "command") {
    const origin = treeView.getPointedTree()?.tree.origin as Element;
    gotoCommandResultAnswer(origin.uri, answer);
    return;
  }
  const demoInfo = getDemoInfoForTreeView(demosManager, treeView);
  if (!demoInfo) {
    return;
  }
  const [demonstration, feedback, editor] = demoInfo;
  if (!(answer in feedback.answer_refs)) {
    // Some answers are generated on the fly by the demo interpreter and do
    // not have counterparts in the demo file.
    return;
  }
  const [queryIdx, answerIdx] = feedback.answer_refs[answer];
  const range = demonstration.queries[queryIdx].__loc_items__answers[answerIdx];
  editor.selection = new vscode.Selection(range.start, range.end);
  editor.revealRange(range);
}

function viewNode(node_id: TraceNodeId, treeView: TreeView) {
  treeView.setSelectedNode(node_id);
}

/////
// Suggest query argument edits
/////

async function updateQueryArgsFromTreeView(
  query: Query,
  demosManager: DemosManager,
  treeView: TreeView,
) {
  const demoInfo = getDemoInfoForTreeView(demosManager, treeView);
  if (!demoInfo) {
    return;
  }
  const [demonstration, _, editor] = demoInfo;
  // Find all queries in the demo with the same name as `query`.
  const matchingQueries = demonstration.queries.filter(
    (q) => q.query === query.name,
  );
  const possibleEdits = matchingQueries.map((q) => {
    const range = q.__loc__args;
    const parentIndentLevel = q.__loc.start.character / 2;
    const newYamlText = prettyYaml(query.args);
    return replaceYamlValue(range, parentIndentLevel, newYamlText);
  });

  if (possibleEdits.length == 0) {
    return;
  }

  alertIfEditorNotTwoSpaceIndent(editor);
  const customRefactorKind = vscode.CodeActionKind.Refactor.append(
    "delphyne.updateQueryArgs",
  );
  const provider = vscode.languages.registerCodeActionsProvider(
    { scheme: "file", language: "yaml" },
    {
      provideCodeActions(document, range, context, token) {
        const codeAction = new vscode.CodeAction(
          "Update query argument",
          customRefactorKind,
        );
        const workspaceEdit = new vscode.WorkspaceEdit();
        possibleEdits.forEach((edit) => {
          const [editRange, newText] = edit;
          workspaceEdit.replace(document.uri, editRange, newText, {
            needsConfirmation: true,
            label: `Update query "${query.name}" arguments`,
          });
        });
        codeAction.edit = workspaceEdit;
        codeAction.isPreferred = true;
        console.log("Return CA.");
        return [codeAction];
      },
    },
    {
      providedCodeActionKinds: [customRefactorKind],
    },
  );
  console.log("Before register");
  // Execute the code action command to show refactoring preview
  await vscode.commands.executeCommand("editor.action.codeAction", {
    kind: customRefactorKind,
    apply: "first",
    showPreview: true,
  });
  console.log("Dispose");
  // Wait a bit before disposing to ensure the code action is applied
  setTimeout(() => {
    provider.dispose();
  }, 500);
}

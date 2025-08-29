//////
/// VSCode Support for Delphyne
//////

import * as vscode from "vscode";
import * as path from "path";
import { TreeView } from "./tree_view";
import {
  showAlert,
  initLogChannels,
  log,
  logInfo,
  logWarning,
} from "./logging";
import { DelphyneServer } from "./server";
import { DemosManager } from "./demos_manager";
import * as testCommands from "./dev_tests";
import { DemosActionsProvider } from "./demos_actions";
import { ElementsManager } from "./elements_manager";
import { registerTreeViewCommands } from "./tree_view_commands";
import { CommandsManager } from "./commands";
import { autoFold } from "./folding";
import { getWorkspaceRoot } from "./config";

//////
/// Activation code
//////

// To be overriden in the `activate` function
let onDeactivateExtension: () => void = () => {};

export async function activate(context: vscode.ExtensionContext) {
  initLogChannels();
  log.debug("Activating Delphyne extension...");
  const treeView = new TreeView(context);
  const server = new DelphyneServer(context);
  await server.start(false);
  const diagnosticCollection: vscode.DiagnosticCollection =
    vscode.languages.createDiagnosticCollection("delphyne");
  context.subscriptions.push(diagnosticCollection);
  onDeactivateExtension = () => server.kill();
  const demosManager = new DemosManager(diagnosticCollection);
  registerTreeViewCommands(context, treeView, demosManager);
  const commandsManager = new CommandsManager(treeView, server);
  const elementsManager = new ElementsManager(demosManager, treeView);
  elementsManager.registerHooks();
  const demosActions = new DemosActionsProvider(demosManager, treeView, server);
  demosActions.register(context);

  // Events
  const onDidOpenTextDocument = (document: vscode.TextDocument) => {
    if (isDemoFile(document)) {
      demosManager.onDidOpenTextDocument(document);
    }
  };
  vscode.workspace.onDidOpenTextDocument(onDidOpenTextDocument);
  vscode.workspace.onDidCloseTextDocument((document) => {
    if (isDemoFile(document)) {
      demosManager.onDidCloseTextDocument(document);
    }
    diagnosticCollection.delete(document.uri);
  });
  vscode.workspace.onDidChangeTextDocument((event) => {
    if (isDemoFile(event.document)) {
      demosManager.onDidChangeTextDocument(event.document);
    }
  });
  vscode.window.onDidChangeActiveTextEditor((editor) => {
    const demoFile = editor && isDemoFile(editor.document);
    vscode.commands.executeCommand(
      "setContext",
      "delphyne.isDemoFile",
      demoFile,
    );
  });

  // Some editors may be open already when the extension is activated so we have
  // to simulare some `onDidOpenTextDocument` events.
  for (const document of vscode.workspace.textDocuments) {
    onDidOpenTextDocument(document);
  }

  // Dev Test Commands
  testCommands.registerCountingTest(context, server);
  testCommands.registerShowCursorPosition(context, demosManager);

  // Keyboard shortcut to show the views
  context.subscriptions.push(
    vscode.commands.registerCommand("delphyne.showViews", () => {
      vscode.commands.executeCommand(
        "workbench.view.extension.delphyneContainer",
      );
    }),
  );

  // Auto-Fold command
  context.subscriptions.push(
    vscode.commands.registerCommand("delphyne.autoFold", () => {
      autoFold(demosManager);
    }),
  );

  // Start and kill the server
  context.subscriptions.push(
    vscode.commands.registerCommand("delphyne.killServer", () => {
      killServerCommand(server);
    }),
  );

  context.subscriptions.push(
    vscode.commands.registerCommand("delphyne.startServer", async () => {
      await startServerCommand(server);
    }),
  );

  // Show root directory command
  context.subscriptions.push(
    vscode.commands.registerCommand("delphyne.showRootDirectory", () => {
      showRootDirectoryCommand();
    }),
  );
}

export function deactivate() {
  onDeactivateExtension();
}

function isDemoFileUri(uri: vscode.Uri): boolean {
  return uri.fsPath.endsWith(".demo.yaml");
}

function isDemoFile(document: vscode.TextDocument): boolean {
  return isDemoFileUri(document.uri);
}

//////
/// Evaluate Command
//////

async function evaluateCommand(server: DelphyneServer) {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return;
  }
}

function killServerCommand(server: DelphyneServer) {
  // Show status bar message while killing
  const loadingStatusBar = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left,
  );
  loadingStatusBar.text = "$(loading~spin) Killing Delphyne server...";
  loadingStatusBar.show();
  const killResult = server.kill();
  // Hide the loading message now that kill operation is complete
  loadingStatusBar.hide();
  loadingStatusBar.dispose();

  const killCommand = '"sudo kill -9 $(sudo lsof -t -i :3008)"';
  if (killResult === undefined) {
    logWarning(
      "No server instance is managed by the Delphyne extension. " +
        `Is an external instance running? If so, consider killing it using ${killCommand}.`,
      true,
    );
  } else if (killResult === false) {
    showAlert(
      "The Delphyne server could not be killed.\n" +
        `Consider running ${killCommand}.`,
    );
  } else {
    logInfo("Delphyne server killed.", true);
  }
}

async function startServerCommand(server: DelphyneServer) {
  await server.start(true);
}

function showRootDirectoryCommand() {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showInformationMessage("No file is currently open.");
    return;
  }

  const filePath = editor.document.fileName;
  const rootDir = getWorkspaceRoot(filePath);

  vscode.window.showInformationMessage(`Root directory: ${rootDir}`);
}

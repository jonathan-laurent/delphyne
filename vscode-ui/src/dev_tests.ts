//////
/// Developer Test Commands
//////

import * as vscode from "vscode";
import { DelphyneServer } from "./server";
import { DemosManager } from "./demos_manager";
import { log } from "./logging";

export function registerCountingTest(
  context: vscode.ExtensionContext,
  server: DelphyneServer,
) {
  context.subscriptions.push(
    vscode.commands.registerCommand("delphyne.dev.runTestCommand", async () => {
      const task = server.launchTask(
        { name: "run", arg: "count", origin: null },
        {},
        "count",
        { n: 5 },
      );
      await task.outcome;
    }),
  );
}

export function registerShowCursorPosition(
  context: vscode.ExtensionContext,
  demosManager: DemosManager,
) {
  context.subscriptions.push(
    vscode.commands.registerCommand("delphyne.dev.showCursorPosition", () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        return;
      }
      const cursor = editor.selection.active;
      log.info(
        "Element at cursor position:",
        demosManager.getElementAt(editor.document.uri, cursor),
      );
    }),
  );
}

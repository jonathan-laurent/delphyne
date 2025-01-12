// const editor = vscode.window.activeTextEditor;
// const range = new vscode.Range(10, 0, 20, 0); // Fold lines 10 to 20
// editor.selections = [new vscode.Selection(range.start, range.end)];
// vscode.commands.executeCommand('editor.fold');

import * as vscode from "vscode";
import { DemosManager } from "./demos_manager";
import { isStrategyDemo } from "./common";

function foldRange(
  editor: vscode.TextEditor,
  range: vscode.Range,
  lineAbove: boolean = false, // The line above the range must be folded
) {
  // Determine if the range spans multiple lines
  if (range.start.line !== range.end.line) {
    const pos = lineAbove ? range.start.translate(-1) : range.start;
    editor.selections = [new vscode.Selection(pos, pos)];
    vscode.commands.executeCommand("editor.fold");
    return;
  }
}

// We fold all `args` sections for queries (when the first line ends with a
// colon) and all `answer` fields (when the string spans multiple lines).
// Everything must be expanded around the cursor though.
export function autoFold(manager: DemosManager) {
  // Use the demos manager to extract the current demo file
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    return;
  }
  const cursor = editor.selection.active; // Save the current cursor position
  const uri = editor.document.uri;
  const demos = manager.getParsedDemoFile(uri);
  if (!demos) {
    return;
  }
  for (const demo of demos) {
    if (!isStrategyDemo(demo)) {
      continue;
    }
    for (const query of demo.queries) {
      foldRange(editor, query.__loc__args, true);
      for (const answer of query.answers) {
        foldRange(editor, answer.__loc__answer, false);
      }
    }
  }
  editor.selection = new vscode.Selection(cursor, cursor); // Restore cursor
}

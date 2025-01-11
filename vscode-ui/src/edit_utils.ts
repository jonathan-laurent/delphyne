//////
/// Utilities to edit YAML files
//////

import * as vscode from "vscode";
import { showAlert } from "./logging";

function editorUsesTwoSpaceIndent(editor: vscode.TextEditor): boolean {
  return editor.options.insertSpaces === true && editor.options.tabSize === 2;
}

function indentString(str: string, level: number, indent: string): string {
  let prefix = indent.repeat(level);
  return str
    .split("\n")
    .map((line) => prefix + line)
    .join("\n");
}

// Insert a new element in a YAML list and return the position of the added element
export function insertYamlListElement(
  editor: vscode.TextEditor,
  listRange: vscode.Range,
  originalListEmpty: boolean,
  newYamlElement: string,
  parentIndentLevel: number,
): vscode.Position {
  if (!editorUsesTwoSpaceIndent(editor)) {
    showAlert("Only two-space indentation is supported.");
  }
  const indent = " ".repeat(2);
  const insertPos = listRange.end;
  let toInsert = indentString(
    newYamlElement.trim(),
    parentIndentLevel + 2,
    indent,
  );
  const newStart = indent.repeat(parentIndentLevel + 1) + "- ";
  toInsert = newStart + toInsert.substring(newStart.length);
  toInsert = originalListEmpty ? "\n" + toInsert : toInsert + "\n";
  editor.edit((editBuilder) => {
    if (originalListEmpty) {
      // We have to erase the current empty list that is in the document
      editBuilder.delete(listRange);
    }
    editBuilder.insert(insertPos, toInsert);
  });
  const firstInsertedLine = originalListEmpty
    ? insertPos.line + 1
    : insertPos.line;
  return new vscode.Position(firstInsertedLine, 2 * (parentIndentLevel + 2));
}

export function getEditorForUri(
  uri: vscode.Uri,
): vscode.TextEditor | undefined {
  // Iterate through all visible text editors
  for (const editor of vscode.window.visibleTextEditors) {
    // Check if the document's URI matches the given URI
    if (editor.document.uri.toString() === uri.toString()) {
      return editor;
    }
  }
  // Return undefined if no matching editor is found
  return undefined;
}

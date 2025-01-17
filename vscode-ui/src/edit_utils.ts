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

// Insert a new element in a YAML list and return the position of the added
// element.
//
// TODO: this function was written via trial and error and it might benefit from
// cleaning up the logic.
export function insertYamlListElements(
  editor: vscode.TextEditor,
  listRange: vscode.Range,
  originalListEmpty: boolean,
  newYamlElements: string[],
  parentIndentLevel: number,
): vscode.Position {
  if (!editorUsesTwoSpaceIndent(editor)) {
    showAlert("Only two-space indentation is supported.");
  }
  const indent = " ".repeat(2);
  const insertPos = listRange.end;
  const newPrefix = indent.repeat(parentIndentLevel + 1) + "- ";
  const elements: string[] = [];
  for (const newYamlElement of newYamlElements) {
    let element = indentString(
      newYamlElement.trim(),
      parentIndentLevel + 2,
      indent,
    );
    element = newPrefix + element.substring(newPrefix.length);
    elements.push(element);
  }
  let toInsert = elements.join("\n");
  toInsert = originalListEmpty ? "\n" + toInsert : toInsert + "\n";
  let addedEmptyLine = false;
  editor.edit((editBuilder) => {
    if (originalListEmpty) {
      // We have to erase the current empty list that is in the document
      editBuilder.delete(listRange);
    } else if (insertPos.line == editor.document.lineCount - 1) {
      // There seems to be an edge case where the list is not empty but it is at
      // the very end of the file and there is no empty line after it. In this
      // case, we add an empty line at the end of the file before doing the
      // change.
      const lastLine = editor.document.lineAt(editor.document.lineCount - 1);
      if (lastLine.text.trim() !== "") {
        editBuilder.insert(lastLine.range.end, "\n");
        addedEmptyLine = true;
      }
    }
    editBuilder.insert(insertPos, toInsert);
  });
  const firstInsertedLine =
    originalListEmpty || addedEmptyLine ? insertPos.line + 1 : insertPos.line;
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

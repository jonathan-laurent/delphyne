//////
/// Utilities to edit YAML files
//////

import * as vscode from "vscode";
import { showAlert } from "./logging";

function editorUsesTwoSpaceIndent(editor: vscode.TextEditor): boolean {
  return editor.options.insertSpaces === true && editor.options.tabSize === 2;
}

export function alertIfEditorNotTwoSpaceIndent(editor: vscode.TextEditor) {
  if (!editorUsesTwoSpaceIndent(editor)) {
    showAlert("Only two-space indentation is supported.");
  }
}

function indentString(str: string, level: number, indent: string): string {
  let prefix = indent.repeat(level);
  return str
    .split("\n")
    .map((line) => (line === "" ? line : prefix + line))
    .join("\n");
}

// Insert a new element in a YAML list and return the position of the added
// element.
export function insertYamlListElements(
  editor: vscode.TextEditor,
  listRange: vscode.Range,
  originalListEmpty: boolean,
  newYamlElements: string[],
  parentIndentLevel: number,
): vscode.Position {
  if (newYamlElements.length === 0) {
    return listRange.start;
  }
  alertIfEditorNotTwoSpaceIndent(editor);
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
  // If the next position after the replaced list (listRange.end) is at the
  // start of a line (e.g. the list is not empty and there is a newline
  // character after it), then we add a "\n" before the inserted elements.
  // Otherwise (e.g. the list is empty or is immediately followed by EOF), we
  // add it after.
  const addInitialNewline = listRange.end.character !== 0;
  const toInsert = addInitialNewline
    ? "\n" + elements.join("\n")
    : elements.join("\n") + "\n";
  editor.edit((editBuilder) => {
    if (originalListEmpty) {
      // We have to erase the current empty list that is in the document
      editBuilder.delete(listRange);
    }
    editBuilder.insert(insertPos, toInsert);
  });
  const firstInsertedLine = addInitialNewline
    ? insertPos.line + 1
    : insertPos.line;
  return new vscode.Position(firstInsertedLine, 2 * (parentIndentLevel + 2));
}

// Create an edit that modifies a YAML value at a given range.
// Must only be applied to editors that use 2-space indentation.
export function replaceYamlValue(
  valueRange: vscode.Range,
  parentIndentLevel: number,
  newYamlText: string,
): [vscode.Range, string] {
  const indent = " ".repeat(2);
  const newText = newYamlText.trim();
  const multilineReplacement = newText.includes("\n");
  const multilineSource = valueRange.start.line !== valueRange.end.line;
  let replacement = newText;
  if (multilineReplacement) {
    replacement = indentString(
      replacement,
      parentIndentLevel + 1,
      indent,
    ).trimStart();
  }
  if (multilineSource) {
    replacement = replacement + "\n";
  }
  return [valueRange, replacement];
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

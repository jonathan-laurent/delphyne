import { StrategyDemo } from "./stubs/demos";
import { Query } from "./stubs/feedback";

import * as vscode from "vscode";
import { prettyYaml } from "./yaml_utils";
import { insertYamlListElement } from "./edit_utils";

export function addQuery(
  query: Query,
  demo: StrategyDemo,
  editor: vscode.TextEditor,
  focus: boolean = true,
) {
  const listRange = demo.__loc__queries;
  const originalListEmpty = demo.queries.length === 0;
  const newItem = { query: query.name, args: query.args, answers: [] };
  const newYamlElement = prettyYaml(newItem);
  const parentIndentLevel = 1;
  const newCursorPos = insertYamlListElement(
    editor,
    listRange,
    originalListEmpty,
    newYamlElement,
    parentIndentLevel,
  );
  if (focus) {
    editor.selection = new vscode.Selection(newCursorPos, newCursorPos);
    editor.revealRange(new vscode.Range(newCursorPos, newCursorPos));
  }
}

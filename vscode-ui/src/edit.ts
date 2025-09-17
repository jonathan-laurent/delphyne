import * as vscode from "vscode";
import { Answer, StrategyDemo } from "./stubs/demos";
import { prettyYaml } from "./yaml_utils";
import { insertYamlListElements } from "./edit_utils";

export function addQueries(
  queries: {
    name: string;
    args: Record<string, unknown>;
    answer: Answer | null; // the __loc fields of Answer must not be set.
  }[],
  demo: StrategyDemo,
  editor: vscode.TextEditor,
  focus: boolean = true,
) {
  const listRange = demo.__loc__queries;
  const originalListEmpty = demo.queries.length === 0;
  const newYamlElements: string[] = queries.map((query) => {
    const answers = query.answer ? [query.answer] : [];
    const item = { query: query.name, args: query.args, answers };
    return prettyYaml(item);
  });
  const parentIndentLevel = 1;
  const newCursorPos = insertYamlListElements(
    editor,
    listRange,
    originalListEmpty,
    newYamlElements,
    parentIndentLevel,
  );
  if (focus) {
    editor.selection = new vscode.Selection(newCursorPos, newCursorPos);
    editor.revealRange(new vscode.Range(newCursorPos, newCursorPos));
  }
}

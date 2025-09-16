import * as vscode from 'vscode';

export type TestCommandString = string;

export interface ToolCall {
  tool: string;
  args: Record<string, unknown>;
  __loc: vscode.Range;
  __loc__tool: vscode.Range;
  __loc__args: vscode.Range;
}

export interface Answer {
  answer: string | unknown;
  call?: ToolCall[];
  structured?: "auto" | true;
  mode?: string | null;
  label?: string | null;
  example?: boolean | null;
  tags?: string[];
  justification?: string | null;
  __loc: vscode.Range;
  __loc__answer: vscode.Range;
  __loc__call: vscode.Range;
  __loc_items__call: vscode.Range[];
  __loc__structured: vscode.Range;
  __loc__mode: vscode.Range;
  __loc__label: vscode.Range;
  __loc__example: vscode.Range;
  __loc__tags: vscode.Range;
  __loc_items__tags: vscode.Range[];
  __loc__justification: vscode.Range;
}

export type AnswerSource = CommandResultAnswerSource | DemoAnswerSource;

export interface CommandResultAnswerSource {
  command: string;
  node_ids?: number[] | null;
  queries?: string[] | null;
  __loc: vscode.Range;
  __loc__command: vscode.Range;
  __loc__node_ids: vscode.Range;
  __loc_items__node_ids: vscode.Range[];
  __loc__queries: vscode.Range;
  __loc_items__queries: vscode.Range[];
}

export interface DemoAnswerSource {
  demo: string;
  queries?: string[] | null;
  __loc: vscode.Range;
  __loc__demo: vscode.Range;
  __loc__queries: vscode.Range;
  __loc_items__queries: vscode.Range[];
}

export interface QueryDemo {
  query: string;
  args: Record<string, unknown>;
  answers: Answer[];
  demonstration?: string | null;
  __loc: vscode.Range;
  __loc__query: vscode.Range;
  __loc__args: vscode.Range;
  __loc__answers: vscode.Range;
  __loc_items__answers: vscode.Range[];
  __loc__demonstration: vscode.Range;
}

export interface StrategyDemo {
  strategy: string;
  args: Record<string, unknown>;
  tests: TestCommandString[];
  queries: QueryDemo[];
  using?: AnswerSource[];
  demonstration?: string | null;
  __loc: vscode.Range;
  __loc__strategy: vscode.Range;
  __loc__args: vscode.Range;
  __loc__tests: vscode.Range;
  __loc_items__tests: vscode.Range[];
  __loc__queries: vscode.Range;
  __loc_items__queries: vscode.Range[];
  __loc__using: vscode.Range;
  __loc_items__using: vscode.Range[];
  __loc__demonstration: vscode.Range;
}

export type Demo = QueryDemo | StrategyDemo;

export type DemoFile = Demo[];


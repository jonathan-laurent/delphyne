import * as vscode from 'vscode';

export type TestCommandString = string;

export interface Answer {
  answer: string;
  label?: string | null;
  example?: boolean | null;
  __loc: vscode.Range;
  __loc__answer: vscode.Range;
  __loc__label?: vscode.Range;
}

export interface DemoQuery {
  query: string;
  args: { [key: string]: unknown };
  answers: Answer[];
  __loc: vscode.Range;
  __loc__query: vscode.Range;
  __loc__args: vscode.Range;
  __loc__answers: vscode.Range;
  __loc_items__answers: vscode.Range[];
}

export interface Demonstration {
  strategy: string;
  args: { [key: string]: unknown };
  tests: TestCommandString[];
  queries: DemoQuery[];
  demonstration?: string | null;
  __loc: vscode.Range;
  __loc__strategy: vscode.Range;
  __loc__args: vscode.Range;
  __loc__tests: vscode.Range;
  __loc_items__tests: vscode.Range[];
  __loc__queries: vscode.Range;
  __loc_items__queries: vscode.Range[];
  __loc__demonstration?: vscode.Range;
}

export type DemoFile = Demonstration[];


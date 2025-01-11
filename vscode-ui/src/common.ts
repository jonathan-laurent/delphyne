import { StrategyDemo, QueryDemo, Demo } from "./stubs/demos";


export const ROOT_ID = 1;

export function isStrategyDemo(demo: Demo): demo is StrategyDemo {
  return demo.hasOwnProperty("strategy");
}

export function isQueryDemo(demo: Demo): demo is QueryDemo {
    return demo.hasOwnProperty("query");
  }
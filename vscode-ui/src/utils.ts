//////
/// Typescript utilities
//////

// Similar to asyncio.wait_for in Python
export function waitFor<T>(
  promise: Promise<T>,
  timeout: number,
): Promise<["timeout", Promise<T>] | ["done", T]> {
  return new Promise((resolve) => {
    setTimeout(() => resolve(["timeout", promise]), timeout);
    promise.then((value) => resolve(["done", value]));
  });
}

export function delay(time_ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, time_ms));
}

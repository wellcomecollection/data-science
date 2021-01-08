import { Dispatch, SetStateAction, useEffect, useState } from "react";

const usePersistedState = <T extends unknown>(
  key: string,
  defaultValue: T
): [T, Dispatch<SetStateAction<T>>] => {
  const getInitial = (): T => {
    if (typeof window !== "undefined") {
      const storedItem = localStorage.getItem(key);
      return storedItem ? (JSON.parse(storedItem) as T) : defaultValue;
    } else {
      return defaultValue;
    }
  };
  const [state, setState] = useState<T>(getInitial);
  useEffect(() => {
    localStorage.setItem(key, JSON.stringify(state));
  }, [key, state]);
  return [state, setState];
};

export default usePersistedState;

import React from "react";

const RadioInput = <T extends string>({
  id,
  value,
  group,
  setValue,
}: {
  id: T;
  value: T;
  group: string;
  setValue: (value: T) => void;
}) => (
  <input
    type="radio"
    name={group}
    value={id}
    id={id}
    checked={value === id}
    onChange={() => setValue(id)}
  />
);

export default RadioInput;

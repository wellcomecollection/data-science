export function rgb_to_hex(r: number, g: number, b: number) {
  return (
    "#" +
    [r, g, b]
      .map(function (x) {
        return ("0" + Math.round(x).toString(16)).slice(-2);
      })
      .join("")
  );
}

export function hex_to_rgb(hex: string) {
  const m = hex.match(/^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i);
  const { r, g, b } = m
    ? {
        r: parseInt(m[1], 16),
        g: parseInt(m[2], 16),
        b: parseInt(m[3], 16),
      }
    : {
        r: 0,
        g: 0,
        b: 0,
      };

  return [r, g, b];
}

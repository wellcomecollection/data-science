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

export function rgb_to_embedding(r: number, g: number, b: number) {
  // the embedding is a flattened n*n*n cube,
  const n = process.env.N_BINS ? parseInt(process.env.N_BINS) : 8;

  // the index of the embedding is the index of the colour in the cube
  const r_index = Math.round((r / 255) * (n - 1));
  const g_index = Math.round((g / 255) * (n - 1));
  const b_index = Math.round((b / 255) * (n - 1));

  const embedding_index = b_index + g_index * n + r_index * n * n;

  const embedding = Array(n * n * n).fill(0);
  embedding[embedding_index] = 1;

  return embedding;
}

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
  const n = process.env.EMBEDDING_DIMENSIONALITY
    ? parseInt(process.env.EMBEDDING_DIMENSIONALITY)
    : 8;

  // the index of the embedding is the index of the colour in the cube
  const r_index = Math.round((r / 255) * (n - 1));
  const g_index = Math.round((g / 255) * (n - 1));
  const b_index = Math.round((b / 255) * (n - 1));

  const embedding_index = b_index + g_index * n + r_index * n * n;

  const neighbours = [
    [r_index - 1, g_index, b_index],
    [r_index + 1, g_index, b_index],
    [r_index, g_index - 1, b_index],
    [r_index, g_index + 1, b_index],
    [r_index, g_index, b_index - 1],
    [r_index, g_index, b_index + 1],
  ];

  const corner_neighbours = [
    [r_index - 1, g_index - 1, b_index - 1],
    [r_index - 1, g_index - 1, b_index + 1],
    [r_index - 1, g_index + 1, b_index - 1],
    [r_index - 1, g_index + 1, b_index + 1],
    [r_index + 1, g_index - 1, b_index - 1],
    [r_index + 1, g_index - 1, b_index + 1],
    [r_index + 1, g_index + 1, b_index - 1],
    [r_index + 1, g_index + 1, b_index + 1],
  ];

    const outer_neighbours = [
      [r_index - 1, g_index, b_index],
      [r_index + 1, g_index, b_index],
      [r_index, g_index - 1, b_index],
      [r_index, g_index + 1, b_index],
      [r_index, g_index, b_index - 1],
      [r_index, g_index, b_index + 1],
    ];

  // the embedding should be an array of zeros with length n*n*n,
  // with a blurry distribution around the index
  const embedding = Array(n * n * n).fill(0);
  embedding[embedding_index] = 0.75;

  neighbours.forEach(([r, g, b]) => {
    if (r >= 0 && r < n && g >= 0 && g < n && b >= 0 && b < n) {
      const neighbour_index = b + g * n + r * n * n;
      embedding[neighbour_index] = 0.25;
    }
  });
  
  corner_neighbours.forEach(([r, g, b]) => {
    if (r >= 0 && r < n && g >= 0 && g < n && b >= 0 && b < n) {
      const neighbour_index = b + g * n + r * n * n;
      embedding[neighbour_index] = 0.1;
    }
  });

  outer_neighbours.forEach(([r, g, b]) => {
    if (r >= 0 && r < n && g >= 0 && g < n && b >= 0 && b < n) {
      const neighbour_index = b + g * n + r * n * n;
      embedding[neighbour_index] = 0.05;
    }
  });

  return embedding;
}

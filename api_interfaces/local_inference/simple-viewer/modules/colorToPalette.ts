import { SimilarityModule } from "./index";
import { hex } from "color-convert";

const allBinSizes = [
  [4, 2, 1],
  [6, 4, 3],
  [9, 6, 5],
];
const binMinima = [0, 10 / 256, 10 / 256];
const minBinWidth = 1 / 256;

const colorToPalette: SimilarityModule = (input: string) => {
  let [h, s, v] = hex.hsv.raw(input);
  h /= 360;
  s /= 100;
  v /= 100;
  const [_, satMin, valMin] = binMinima;
  return allBinSizes.map((binSizes, i) => {
    let nextBin = 0;
    if (v < valMin) {
      return `${nextBin}/${i}`;
    }
    nextBin += 1;

    if (s < satMin) {
      return `${
        nextBin +
        Math.floor((binSizes[1] * (v - valMin)) / (1 - valMin + minBinWidth))
      }/${i}`;
    }
    nextBin += binSizes[1];

    const idx = (x: number, i: number) =>
      Math.floor(
        (binSizes[i] * (x - binMinima[i])) / (1 - binMinima[i] + minBinWidth)
      );
    const binIndex =
      nextBin +
      idx(h, 0) +
      binSizes[0] * idx(s, 1) +
      binSizes[0] * binSizes[1] * idx(v, 2);
    return `${binIndex}/${i}`;
  });
};

export default colorToPalette;

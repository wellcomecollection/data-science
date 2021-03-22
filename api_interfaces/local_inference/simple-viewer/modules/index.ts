const modules = {
  csv: import("./csv"),
  identity: import("./identity"),
  colorToPalette: import("./colorToPalette"),
} as const;

export type ModuleName = keyof typeof modules;

export const moduleNames = Object.keys(modules) as Array<ModuleName>;

export type SimilarityModule = (input: string) => string | string[];

export const loadModule = async (
  moduleName: ModuleName
): Promise<SimilarityModule> => {
  const module = await modules[moduleName];
  return module.default;
};

# Outstanding questions

Assorted questions about the scope of the problem and possible solutions, grouped by theme.

## what's interesting?

- which margin discolourations are stains, which ones are historically interesting?
- When is lettering florid enough to be an interesting image?

## end results & search

- when should we extract an interesting section of a page, and when should we extract the whole thing to show the interesting bit in context?
- which text do we include alongside the image to make it searchable (not just discoverable). should we just include a book's title, or try to determine the caption for the image, or take all of the text on the page, or something else?
- should we consider using the DeViSE search method for searching these images?

## pipeline

- if the model changes, the number of interesting images extracted will almost definitely change. are we happy to get rid of the old ones which the model no longer deems interesting? what happens to those links/ids?
- if a 'crop' changes to include/exclude more of the original page, is that a new image? is it the same as the old image?
- does this enricher need to sit by the matcher/merger? does it sit in the catalogue

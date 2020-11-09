# Possible approaches

Assorted approaches to the problem

## multi-stage feature extraction

- VGG (or similar) feature extraction applied to every page
- then, the core model might look like an autoencoder. Train on many many many pages of digitised books, and use like an anomaly detector to pick out any pages which look interesting
- subsequent model (yolo or some kind of similar attention-based model) to pick out interesting regions of the page.
- each of tehse regions might have VGG applied to it to generate a feature vector for its own similarity hash

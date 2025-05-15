# RingRodeo Desktop

This repository contains a method to run the RingRodeo web application from our paper [Domain-specific AI segmentation of IMPDH2 rod/ring structures in mouse embryonic stem cells](https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-025-02226-7) for automatic segmentation of rod-ring structures in IMPDH2 stem cells.

## Installation

To run the webapp, you will need [Docker](https://docker.com)

To run the webapp, run:

```bash
docker compose build && docker compose up
```

This will spin up the Python Flask backend API (that performs the segmentation) as well as the frontend React GUI on your computer.

> The segmentation is really slow/crashes my machine!

The model attempts to find a GPU on your computer - if not, it uses the CPU which might use significant resources (it may crash with <8GB RAM). In short - it's deep learning; it needs a pretty beefy computer!

> Help! I get an error *and* it says the API exited with code 137

This is because you need to give Docker more RAM. Go into the Docker desktop App > Settings > resources and turn up the memory!

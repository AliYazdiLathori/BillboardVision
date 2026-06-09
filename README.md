# HighViewDet
This project focuses on evaluating and improving human detection models for real-world surveillance scenarios.

The project has two main objectives:

Model Benchmarking: Develop a robust evaluation pipeline for testing different object detection models on video data. So far, the project includes several one-stage detectors, primarily YOLO-based models. Future work will extend the benchmark to include two-stage detectors and compare their performance under identical conditions.
Domain Adaptation for High-Angle Surveillance: Improve the performance of human detection models for high-angle camera views, such as cameras installed on top of billboards or other elevated structures. A particular focus is improving detection robustness across diverse clothing styles commonly encountered in Islamic regions, including various forms of veiled clothing that may be underrepresented in the datasets used to train the original models.

The repository also contains videos captured from high-angle viewpoints, extracted video frames, corresponding annotation files, and benchmark results obtained using pretrained models. Fine-tuned models and updated evaluation results will be added as the project progresses.

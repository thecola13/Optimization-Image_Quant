# Optimization-Image_Quant
Assignment for "20874 - Algorithms for Optimisation and Inference" course at Bocconi University (MSc in Artificial Intelligence)

Run these commands to use the various files

- Point 1: python3 recolor.py {input/image/path} {output/image/path} {k}

- Point 3.a: python3 datagen.py {input/image/path} {output/datafile/path} {k} {threshold}
    Suggested threshold: 280M
    glpsol finds optimal solution with thresholds up to 800M and down to 60M for 20col.png, k = 8

- Point 3.b: glpsol --math problem.mod --data {datafile/path}
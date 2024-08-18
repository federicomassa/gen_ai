import numpy as np
import matplotlib.pyplot as plt
import math
from gen_ai.models import simple_dense_1
from tensorflow.keras import optimizers, utils, datasets
from notebooks.utils import display


def my_print():
    ## 0. Parameters <a name="parameters"></a>
    NUM_CLASSES = 10
    ## 1. Prepare the Data <a name="prepare"></a>
    (x_1, y_1), (x_2, y_2) = datasets.cifar10.load_data()
    x_tot = np.concatenate((x_1, x_2), axis=0)
    y_tot = np.concatenate((y_1, y_2), axis=0)

    train_test_ratio = float(1.0 / 6.0)
    sample_size = x_tot.shape[0]
    print(sample_size)
    split_idx = math.ceil(sample_size * train_test_ratio)

    x_train = x_tot[:split_idx]
    y_train = y_tot[:split_idx]

    x_test = x_tot[split_idx:]
    y_test = y_tot[split_idx:]

    print(x_train.shape)
    print(y_train.shape)

    print(x_test.shape)
    print(y_test.shape)
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train = utils.to_categorical(y_train, NUM_CLASSES)
    y_test = utils.to_categorical(y_test, NUM_CLASSES)
    display(x_train[:10])
    print(y_train[:10])

    model_config = simple_dense_1.Config(
        input_shape=(32, 32, 3),
        dense_layer_sizes=[200, 150],
        dense_layer_activation="relu",
        output_classes=NUM_CLASSES,
        output_activation="softmax",
    )
    model = simple_dense_1.make_model(model_config)

    model.summary()
    ## 3. Train the model <a name="train"></a>
    opt = optimizers.Adam(learning_rate=0.0005)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=32, epochs=20, shuffle=True)
    ## 4. Evaluation <a name="evaluate"></a>
    model.evaluate(x_test, y_test)
    CLASSES = np.array(
        [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
    )

    preds = model.predict(x_train)
    preds_single = CLASSES[np.argmax(preds, axis=-1)]
    actual_single = CLASSES[np.argmax(y_train, axis=-1)]
    n_to_show = 10
    indices = np.random.choice(range(len(x_train)), n_to_show)

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i, idx in enumerate(indices):
        img = x_train[idx]
        ax = fig.add_subplot(1, n_to_show, i + 1)
        ax.axis("off")
        ax.text(
            0.5,
            -0.35,
            "pred = " + str(preds_single[idx]),
            fontsize=10,
            ha="center",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            -0.7,
            "act = " + str(actual_single[idx]),
            fontsize=10,
            ha="center",
            transform=ax.transAxes,
        )
        ax.imshow(img)

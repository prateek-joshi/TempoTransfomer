import sys
sys.path.append("E:\\TempoTransformer")

import os
import tensorflow as tf
import tensorflow_addons as tfa
from argparse import ArgumentParser
from Temformer.data import TempoDataset
from Temformer.model import VisionTransformer

AUTOTUNE = tf.data.experimental.AUTOTUNE

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--split", default=0.1, type=float)
    parser.add_argument("--image-size", default=32, type=int)
    parser.add_argument("--patch-size", default=4, type=int)
    parser.add_argument("--num-layers", default=4, type=int)
    parser.add_argument("--d-model", default=64, type=int)
    parser.add_argument("--num-heads", default=4, type=int)
    parser.add_argument("--mlp-dim", default=128, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=4096, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    args = parser.parse_args()

    MANIFEST_PATH = 'E:\\TempoTransformer\\data\\final_manifest_new.csv'
    d = TempoDataset(MANIFEST_PATH, val_split=args.split,n_fft=2048)
    train,val = d.get_spectrogram_generator()

    train = train.batch(args.batch_size).prefetch(AUTOTUNE)
    val = val.batch(args.batch_size).prefetch(AUTOTUNE)

    model = VisionTransformer(
            image_size=args.image_size,
            patch_size=args.patch_size,
            num_layers=args.num_layers,
            d_model=args.d_model,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            channels=1,
            dropout=0.1,
        )
    model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=tfa.optimizers.AdamW(
                learning_rate=args.lr, weight_decay=args.weight_decay
            ),
        )

    model.build((None, 128, 128, 1))
    print(model.summary())
    cbs = [
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(args.logdir,'cp.ckpt'), save_weights_only=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=args.logdir, profile_batch=0)
    ]
    model.fit(
        train,
        validation_data=val,
        epochs=args.epochs,
        callbacks=cbs
    )
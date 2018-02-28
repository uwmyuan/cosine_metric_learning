# vim: expandtab:ts=4:sw=4
import functools
import os
import numpy as np
import scipy.io as sio
import train_app
from datasets import sdd
from datasets import util
import nets.deep_sort.network_definition as net


class sdd_dataset(object):

    def __init__(self, dataset_dir, annotation_file_name,video_file_name,num_validation_y=0.1, seed=1234):
        self._dataset_dir = dataset_dir
        self._num_validation_y = num_validation_y
        self._seed = seed
        self._annotation_file_name=annotation_file_name
        self._video_file_name=video_file_name

    def read_train(self):

        filenames, ids, _ = sdd.read_train_split_to_str(
            self._dataset_dir+self._video_file_name,self._dataset_dir+self._annotation_file_name)
        train_indices, _ = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        filenames = [filenames[i] for i in train_indices]
        ids = [ids[i] for i in train_indices]
        #camera_indices = [camera_indices[i] for i in train_indices]
        camera_indices = np.zeros(len(ids))
        return filenames, ids, camera_indices

    def read_validation(self):
        filenames, ids, _ = sdd.read_train_split_to_str(
            self._dataset_dir+self._video_file_name,self._dataset_dir+self._annotation_file_name)
        _, valid_indices = util.create_validation_split(
            np.asarray(ids, np.int64), self._num_validation_y, self._seed)

        filenames = [filenames[i] for i in valid_indices]
        ids = [ids[i] for i in valid_indices]
        #camera_indices = [camera_indices[i] for i in valid_indices]
        camera_indices = np.zeros(len(ids))
        return filenames, ids, camera_indices

    def read_test(self):
        return sdd.read_test_split_to_str(self._dataset_dir)

import time,sys
def main():
    # output to file
    this_time = time.strftime('%Y-%m-%dT%H%M%S')
    sys.stdout = open(this_time + 'sdd.txt', 'w')
    sys.stderr = open(this_time + 'sdd.log', 'w')
    arg_parser = train_app.create_default_argument_parser("sdd")
    arg_parser.add_argument(
        "--dataset_dir", help="Path to Stanford drone dataset directory.",
        default="../StanfordDroneDataset")
    arg_parser.add_argument(
        "--sdk_dir", help="Path to sdd baseline evaluation software.",
        default="resources/StanfordDroneDataset-baseline")
    arg_parser.add_argument(
        "--annotation_file_name", help="Path to Stanford drone dataset annotation file.",
        default="/annotations/nexus/video0/annotations.txt")
    arg_parser.add_argument(
        "--video_file_name", help="Path to Stanford drone dataset video file.",
        default="/videos/nexus/video0/video.mov")
    args = arg_parser.parse_args()
    dataset = sdd_dataset(args.dataset_dir,args.annotation_file_name,args.video_file_name,num_validation_y=0.1, seed=1234)

    if args.mode == "train":
        train_x, train_y, _ = dataset.read_train()
        print("Train set size: %d images, %d identities" % (
            len(train_x), len(np.unique(train_y))))
        network_factory = net.create_network_factory(
            is_training=True, num_classes=sdd.calculate_max_label(dataset._dataset_dir+dataset._annotation_file_name) + 1,
            add_logits=args.loss_mode == "cosine-softmax")
        train_kwargs = train_app.to_train_kwargs(args)
        train_app.train_loop(
            net.preprocess, network_factory, train_x, train_y,
            num_images_per_id=4, image_shape=sdd.IMAGE_SHAPE,
            **train_kwargs)
    elif args.mode == "eval":
        valid_x, valid_y, camera_indices = dataset.read_validation()
        print("Validation set size: %d images, %d identities" % (
            len(valid_x), len(np.unique(valid_y))))

        network_factory = net.create_network_factory(
            is_training=False, num_classes=sdd.calculate_max_label(self._dataset_dir+self._annotation_file_name) + 1,
            add_logits=args.loss_mode == "cosine-softmax")
        eval_kwargs = train_app.to_eval_kwargs(args)
        train_app.eval_loop(
            net.preprocess, network_factory, valid_x, valid_y, camera_indices,
            image_shape=sdd.IMAGE_SHAPE, **eval_kwargs)
    elif args.mode == "export":
        # Export one specific model.
        gallery_filenames, _, query_filenames, _, _ = dataset.read_test()

        network_factory = net.create_network_factory(
            is_training=False, num_classes=sdd.calculate_max_label(self._dataset_dir+self._annotation_file_name) + 1,
            add_logits=False, reuse=None)
        gallery_features = train_app.encode(
            net.preprocess, network_factory, args.restore_path,
            gallery_filenames, image_shape=sdd.IMAGE_SHAPE)
        sio.savemat(
            os.path.join(args.sdk_dir, "feat_test.mat"),
            {"features": gallery_features})

        network_factory = net.create_network_factory(
            is_training=False, num_classes=sdd.calculate_max_label(self._dataset_dir+self._annotation_file_name) + 1,
            add_logits=False, reuse=True)
        query_features = train_app.encode(
            net.preprocess, network_factory, args.restore_path,
            query_filenames, image_shape=sdd.IMAGE_SHAPE)
        sio.savemat(
            os.path.join(args.sdk_dir, "feat_query.mat"),
            {"features": query_features})
    elif args.mode == "finalize":
        network_factory = net.create_network_factory(
            is_training=False, num_classes=sdd.calculate_max_label(self._dataset_dir+self._annotation_file_name) + 1,
            add_logits=False, reuse=None)
        train_app.finalize(
            functools.partial(net.preprocess, input_is_bgr=True),
            network_factory, args.restore_path,
            image_shape=sdd.IMAGE_SHAPE,
            output_filename="./sdd.ckpt")
    elif args.mode == "freeze":
        network_factory = net.create_network_factory(
            is_training=False, num_classes=sdd.calculate_max_label(self._dataset_dir+self._annotation_file_name) + 1,
            add_logits=False, reuse=None)
        train_app.freeze(
            functools.partial(net.preprocess, input_is_bgr=True),
            network_factory, args.restore_path,
            image_shape=sdd.IMAGE_SHAPE,
            output_filename="./sdd.pb")
    else:
        raise ValueError("Invalid mode argument.")


if __name__ == "__main__":
    main()

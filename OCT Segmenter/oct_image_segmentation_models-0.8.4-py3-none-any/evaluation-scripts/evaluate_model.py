import sys

from pathlib import Path

file = Path(__file__).resolve()
package_root_directory = file.parents[1]
sys.path.append(str(package_root_directory))

from oct_image_segmentation_models.common import augmentation as aug
from oct_image_segmentation_models import evaluation
from oct_image_segmentation_models.model import (
    evaluation_parameters as eparams,
)
from oct_image_segmentation_models.model import save_parameters


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python3 evaluate_model.py <path/to/model/file> "
            "</path/to/test/dataset/file> </output/dir/path>"
        )
        exit(1)

    model_file_path = Path(sys.argv[1])
    test_dataset_path = Path(sys.argv[2])
    results_dir = Path(sys.argv[3])

    save_params = save_parameters.SaveParameters(
        pngimages=True,
        raw_image=True,
        raw_labels=True,
        temp_extra=True,
        boundary_maps=True,
        area_maps=True,
        comb_area_maps=True,
        seg_plot=True,
    )

    eval_params = eparams.EvaluationParameters(
        model_file_path=model_file_path,
        dataset_file_path=test_dataset_path,
        is_evaluate=True,
        col_error_range=None,
        save_foldername=results_dir.absolute(),
        eval_mode="both",
        aug_fn_arg=(aug.no_aug, {}),
        save_params=save_params,
        verbosity=3,
        gsgrad=1,
        transpose=False,
        normalise_input=True,
        comb_pred=False,
        recalc_errors=False,
        boundaries=False,
        trim_maps=False,
        trim_ref_ind=0,
        trim_window=(0, 0),
        dice_errors=True,
        flatten_image=False,
        flatten_ind=0,
        flatten_poly=False,
        binarize=True,
        binarize_after=True,
        bg_ilm=True,
        bg_csi=False,
        flatten_pred_edges=False,
        flat_marg=0,
        use_thresh=False,
        thresh=0.5,
    )

    evaluation.evaluate_model(eval_params)

from collections import Counter
from pathlib import Path

import encord
import numpy as np


def fetch_client(encord_ssh_key_path: Path):
    """
    Create an Encord client using an SSH private key.

    Parameters
    ----------
    encord_ssh_key_path
        Path to the SSH key registered with Encord.

    Returns
    -------
    encord.EncordUserClient
        Authenticated client instance.
    """
    return encord.EncordUserClient.create_with_ssh_private_key(
        ssh_private_key_path=encord_ssh_key_path
    )


def fetch_project(client: encord.EncordUserClient, encord_project_hash: str):
    """
    Retrieve an Encord project.

    Parameters
    ----------
    client
        Authenticated Encord client.
    encord_project_hash
        Public string hash of the target project.

    Returns
    -------
    encord.project.Project
        The requested project object.
    """
    return client.get_project(encord_project_hash)


def fetch_dataset(client: encord.EncordUserClient, encord_dataset_hash: str):
    """
    Retrieve an Encord dataset.

    Parameters
    ----------
    client
        Authenticated Encord client.
    encord_dataset_hash
        Public hash of the target dataset.

    Returns
    -------
    encord.dataset.Dataset
        The requested dataset object.
    """
    return client.get_dataset(encord_dataset_hash)


def fetch_annotations_from_workflow_stages(project: encord.project.Project):
    """
    Collect task titles (removing ``.png`` suffix) for every workflow stage.

    Parameters
    ----------
    project
        Project whose workflow stages are inspected.

    Returns
    -------
    dict[str, list[str]]
        Mapping ``{stage_title: [task_title1, task_title2, …]}``.
    """
    annotations = {
        stage.title: [
            str(task.data_title).replace('.png', '')
            for task in stage.get_tasks()
        ]
        for stage in project.workflow.stages
    }
    return annotations


def fetch_annotations_duplicates(
    annotations: dict[str, list[str]],
) -> list[str]:
    """
    Identify duplicate annotation IDs inside the same stage.

    Parameters
    ----------
    annotations:
        Mapping ``{stage_name: [annotation_id, ...]}``.

    Returns
    -------
    list[str]
        Sorted list of annotation IDs that occur more than once
        in at least one stage.  Each ID appears only once in the output.

    Examples
    --------
    >>> anns = {"Annotate 1": ["a1", "a2", "a1"], "Complete": ["b1"]}
    >>> fetch_annotations_duplicates(anns)
    ['a1']
    """
    return sorted(
        {
            ann_id
            for ids in annotations.values()
            for ann_id, cnt in Counter(ids).items()
            if cnt > 1
        }
    )


def fetch_annotation_signed_url(
    encord_client: encord.EncordUserClient, product_name: str
) -> str | None:
    """
    Fetch the signed URL of the first remote annotation that matches the
    provided *product_name*.

    The function calls :py:meth:`encord.EncordUserClient.find_storage_items`
    and returns the signed URL of the first item whose file name contains
    ``"_rgb"``.  If no such item exists, ``None`` is returned.

    Parameters
    ----------
    encord_client : encord.EncordUserClient
        A logged-in Encord client used to query storage items.
    product_name : str
        The product identifier (or prefix) used as the search term.

    Returns
    -------
    str | None
        Signed URL of the matching annotation, or ``None`` when no match is
        found.

    Examples
    --------
    >>> client = encord.EncordUserClient.from_...
    >>> url = fetch_annotation_signed_url(client, "hyperfield1a_L1B_20250329T011705")
    >>> if url:
    ...     print("Found annotation:", url)
    ... else:
    ...     print("Annotation not found")
    """
    return next(
        (
            item.get_signed_url()
            for item in encord_client.find_storage_items(search=product_name)
            if '_rgb' in item.name
        ),
        None,
    )


def get_annotation_tensor(
    prod_name: str,
    encord_project: encord.project.Project,
    label_map: dict[str, int] | None = None,
) -> np.ndarray | None:
    """
    Build a 2-D mask from the bit-mask annotations stored in an Encord
    project.

    Each pixel in the returned array is filled with the value specified in
    *label_map* for the corresponding mask type (class); pixels that are not
    covered by any annotation keep the value of the background class
    (default ``0``, a.k.a. “Fill”).

    Parameters
    ----------
    prod_name :
        Title of the image (frame) as it appears in the Encord project.
        If no label row is found, an additional lookup with
        ``prod_name + ".png"`` is attempted for backward compatibility.
    encord_project :
        An initialised Encord Project instance used to access label rows
        and masks.
    label_map :
        Mapping ``{class_name: pixel_value}`` that defines the greyscale
        intensity assigned to each annotation class.  When *None*,
        the following default is used::

            {
                "Fill":          0,
                "Cloud_Shadow": 64,
                "Clear":       128,
                "Thin_Cloud":  192,
                "Cloud":       255,
            }

    Returns
    -------
    numpy.ndarray or None
        A 2-D ``uint8`` array containing the merged mask, or ``None`` when
        the image is present but has no bit-mask annotation.

    Raises
    ------
    FileNotFoundError
        If no label row is found for *prod_name* (or its “*.png*”
        variant).
    """
    # Considering cloud detection as base case
    if label_map is None:
        label_map = {
            'Fill': 0,
            'Cloud_Shadow': 64,
            'Clear': 128,
            'Thin_Cloud': 192,
            'Cloud': 255,
        }
    label_rows = encord_project.list_label_rows_v2(data_title_eq=prod_name)
    if len(label_rows) == 0:
        # Retry considering old name
        label_rows = encord_project.list_label_rows_v2(
            data_title_eq=prod_name + '.png'
        )
    if len(label_rows) == 0:
        return None

    label_row = label_rows[0]
    label_row.initialise_labels()

    # Get object instances for the specific file
    # The annotations should always be on the frame contianing RGB bands,
    # however to prevent wrong indexing is better to look up for all the
    # frames until a label is found
    for frame_idx in range(0, label_row.number_of_frames):
        object_instances = label_row.get_frame_view(
            frame_idx
        ).get_object_instances()
        num_instances = len(object_instances)
        if num_instances != 0:
            break

    # Annotation not found, returning None
    if num_instances == 0:
        return None

    for i in range(num_instances):
        object_instance = object_instances[i]
        attr = object_instance.ontology_item.attributes[0]
        mask_type = object_instance.get_answer(attr).value
        bitmask_annotation = object_instance.get_annotations()[0]
        bitmask = bitmask_annotation.coordinates.to_numpy_array().astype(
            np.uint8
        )
        break

    # Force combined mask to be good
    combined_mask = np.zeros(
        (bitmask.shape[0], bitmask.shape[1]), dtype=np.uint8
    )

    for i in range(num_instances):
        object_instance = object_instances[i]
        attr = object_instance.ontology_item.attributes[0]
        mask_type = object_instance.get_answer(attr).value
        bitmask_annotation = object_instance.get_annotations()[0]
        bitmask = bitmask_annotation.coordinates.to_numpy_array().astype(
            np.uint8
        )

        # Assign intensity values based on mask type
        if mask_type in label_map:
            intensity = label_map[mask_type]
            combined_mask[bitmask == 1] = intensity

    # NOTE: No data value should be added considering the latest product generated by the pipeline
    return combined_mask

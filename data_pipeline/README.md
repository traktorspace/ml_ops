# Cloud Annotations management

## Scripts available
| Notebook/Script                    | Usage                                                                                                                  |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `encord/read_encord_dataset.ipynb` | Read dataset from Encord and store to file the list of product that need to be moved to `annotations_completed` folder |
| XXX                                | XXX                                                                                                                    |

![annotation-processes](media/cloud_annotation_process.png)

# Setup
https://cloud.google.com/docs/authentication/gcloud

Be sure that you have access to all the buckets you need. 
For annotation-related buckets please contact Tommaso Canova or Arthur Vandenhoeke, for products-only-related buckets please contact Lennert Antson or Matti Bragge.

```
gcloud auth application-default login
```
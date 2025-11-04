../apps/utils/compute_groundtruth --data_type float --dist_fn l2 --base_file siftsmall/siftsmall_base.bin --query_file siftsmall/siftsmall_query.bin --gt_file siftsmall/siftsmall_gt_35.bin --K 100 --label_file ./rand_labels_50_10K.txt --filter_label 35 --universal_label 0
../apps/build_memory_index  --data_type float --dist_fn l2 --data_path siftsmall/siftsmall_base.bin --index_path_prefix siftsmall/siftsmall_R32_L50_filtered_index -R 32 --FilteredLbuild 50 --alpha 1.2 --label_file ./rand_labels_50_10K.txt --universal_label 0
../apps/build_stitched_index --data_type float --data_path siftsmall/siftsmall_base.bin --index_path_prefix siftsmall/siftsmall_R20_L40_SR32_stitched_index -R 20 -L 40 --stitched_R 32 --alpha 1.2 --label_file ./rand_labels_50_10K.txt --universal_label 0


../apps/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix siftsmall/siftsmall_R32_L50_filtered_index --query_file siftsmall/siftsmall_query.bin --gt_file siftsmall/siftsmall_gt_35.bin --filter_label 35 -K 10 -L 10 20 30 40 50 100 --result_path siftsmall/filtered_search_results


../apps/search_memory_index  --data_type float --dist_fn l2 --index_path_prefix siftsmall/siftsmall_R20_L40_SR32_stitched_index --query_file siftsmall/siftsmall_query.bin --gt_file siftsmall/siftsmall_gt_35.bin --filter_label 35 -K 10 -L 10 20 30 40 50 100 --result_path siftsmall/stitched_search_results
2\_build\_matrix
================

# Goal

For each AST, create a matrix where the rows are the block number the
student chose, the columns are the block types, and in each row there is
a 1 representing the block that student chose.

The number of rows in this matrix needs to be equal to the longest AST
we have, so there is an additional block type “blank” which means that
AST already ended.

For example, if an AST included a lone block of move\_forward, and the
longest AST included only 2 blocks, then the matrix would be the
following:

``` r
library(tidyverse)

tribble(
  ~move_forward, ~turn_left, ~turn_right, ~blank,
       1,             0,           0,        0,
       0,             0,           0,        1
)
```

    ## # A tibble: 2 x 4
    ##   move_forward turn_left turn_right blank
    ##          <dbl>     <dbl>      <dbl> <dbl>
    ## 1            1         0          0     0
    ## 2            0         0          0     1

# Write a function

Load in data.

``` r
trees_4 <- 
    read_rds(here::here("data-created", "trees_4.rds")) %>% 
    filter(tree %>% map_lgl(~ "type" %in% names(.)))
```

Look at all trees to get dimensions of matrix.

``` r
# need number of blocks for rows of matrix
num_rows <- map_int(trees_4$tree, nrow) %>% max()

# need all types for columns of matrix
all_types <-
    trees_4$tree %>%
    map(~ distinct(., type)) %>%
    bind_rows() %>%
    count(type, sort = TRUE) %>%
    mutate(percent_of_trees = n / nrow(trees_4))

num_cols <- nrow(all_types) + 1 # add 1 for blank
```

Write a function that takes the sequence of blocks, num\_cols, and
num\_rows of the output matrix and creates the matrix.

``` r
ast_blocks_to_matrix <- function(blocks, num_cols, num_rows){
    
    block_names <- c("maze_moveForward", "turnLeft", "turnRight", "blank")

    where_one_is_in_each_row <-
        blocks %>%
        c(rep("blank", num_rows - length(blocks))) %>%
        parse_factor(levels = block_names) %>%
        as.integer()
    
    ast_df_as_vector <- rep(0, num_rows * num_cols)
    
    ast_df_as_vector[where_one_is_in_each_row + 4 * (1:num_rows - 1)] <- 1
    
    ast_df_as_vector %>%
        matrix(ncol = num_cols, byrow = TRUE) %>%
        as_tibble(.name_repair = "minimal") %>%
        set_names(block_names)
}
```

# Use that function

``` r
list_of_ast_matrices <- 
    trees_4$tree %>% 
    map(~ .$type) %>% 
    map(~ ast_blocks_to_matrix(., num_cols, num_rows)) %>% 
    map(as.matrix)

array_of_ast_matrices <- 
    list_of_ast_matrices %>% 
    unlist() %>% 
    array(dim = c(num_rows, num_cols, length(list_of_ast_matrices)))
```

# Output results

Transfer to python

``` r
library(reticulate)
py$array_of_ast_matrices <- r_to_py(array_of_ast_matrices)
```

Save as .npy file using Python

``` python
import numpy as np
np.save("../data-created/q4_array_of_ast_matrices.npy", array_of_ast_matrices)
# np.load("data-created/q4_array_of_ast_matrices.npy")
```

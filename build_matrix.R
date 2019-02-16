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

# write a function that turns a list into a matrix
t <- trees_4$tree[[1]]

block_names <- c("maze_moveForward", "turnLeft", "turnRight", "blank")

where_one_is_in_each_row <-
    t$type %>%
    c(rep("blank", num_rows - nrow(t))) %>%
    as_factor(block_sequence, levels = block_names) %>%
    as.integer()

ast_df_as_vector <- rep(0, num_rows * num_cols)

ast_df_as_vector[where_one_is_in_each_row + 4 * (1:num_rows - 1)] <- 1

ast_df_as_vector %>%
    matrix(ncol = num_cols, byrow = TRUE) %>%
    as_tibble() %>%
    set_names(block_names)



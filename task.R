out <-
    trees_4$tree %>%
    map(~ distinct(., type)) %>%
    bind_rows() %>%
    count(type, sort = TRUE) %>%
    mutate(percent_of_trees = n / nrow(trees_4))

trees_4 %>% nrow()

out

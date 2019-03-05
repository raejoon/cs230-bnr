library(tidyverse)

# load what we have -------------------------------------------------------
scores <- read_tsv("data-raw/hoc4/asts/unitTestResults.txt") %>% arrange(desc(score))

trees_4 <- read_rds(here::here("data-created", "trees_4.rds")) %>% filter(tree %>% map_lgl(~ "type" %in% names(.)))

id_to_trajectory <- read_csv("data-raw/hoc4/trajectories/idMap.txt")

trajectories <- 
    tibble(trajectoryId = list.files("data-raw/hoc4/trajectories")) %>% 
    mutate(
        asts = trajectoryId %>% map(~ read_lines(glue::glue("data-raw/hoc4/trajectories/{.}"))),
        asts = asts %>% map(as.numeric),
        trajectoryId = parse_number(trajectoryId)
    ) %>% 
    filter(!is.na(trajectoryId))

correct_asts <- filter(scores, score == 100)$astId
# one student path --------------------------------------------------------
# the 7th stuent had trajectory id 17...
id_to_trajectory %>% count(trajectoryId)

# this means they went through these asts in this order
trajectories$asts[trajectories$trajectoryId == 17]

trajectories %>% mutate(count = asts %>% map_int(length)) %>% arrange(desc(count))

tmp <- 
    trajectories %>% 
    mutate(count_correct = asts %>% map_int(~ sum(. %in% correct_asts)))

tmp %>% arrange(desc(count_correct))

filter(trajectories, trajectoryId == 10006)$asts[[1]] %in% correct_asts

filter(trajectories, trajectoryId == 10006)$asts[[1]]

# work at the trajectory level --------------------------------------------
# write a function
ast_to_correct_within_k <- function(ast, k){
    where_correct <- ast %in% correct_asts
    lead_correct <- lead(where_correct, k)
    out <- numeric(0)
    if (!is.na(lead_correct[1])){
        out <- lead_correct %>% replace_na(0) %>% cumsum()
    } else(
        out <- lead_correct %>% replace_na(1)
    )
    out
}

trajectory_correct_within_k <- 
    trajectories %>% 
    mutate(
        correct_within_k = asts %>% map(ast_to_correct_within_k, k = 1),
        correct_within_k_padded = 
            correct_within_k %>% map(
                ~ c(., rep(last(.), max(map_int(correct_within_k, length)) - length(.)))
            )
    ) %>% 
    select(trajectoryId, correct_within_k_padded)

attempt_matrix <- 
    id_to_trajectory %>% 
    left_join(trajectory_correct_within_k, by = "trajectoryId") %>% 
    #filter(!correct_within_k_padded %>% map_lgl(is.null)) %>%  # will need to remove
    select(-trajectoryId) %>% 
    unnest() %>% 
    group_by(secretId) %>% 
    mutate(attempt = row_number()) %>% 
    ungroup() %>% 
    spread(attempt, correct_within_k_padded)

attempt_matrix <- # trajectory 71 was the bad apple because it had 0 asts
    attempt_matrix %>% 
    filter(!is.na(`1`))

attempt_matrix %>% write_csv("~/Desktop/correct_within_1.csv")

attempt_matrix %>% colMeans()

# error check -------------------------------------------------------------
bad <- attempt_matrix %>% filter(is.na(`1`)) %>% pull(secretId)

id_to_trajectory %>% filter(secretId %in% bad) %>% count(trajectoryId)

trajectories %>% 
    filter(trajectoryId == 71)








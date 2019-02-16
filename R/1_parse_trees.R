library(tidyverse)
library(magrittr)
library(jsonlite)
source("functions.R")

# problem 4 ---------------------------------------------------------------
trees_4 <-
    tibble(
        file = list.files(here::here("data-raw", "hoc4", "asts")) %>% str_subset(".json$")
    ) %>%
    mutate(
        astId = str_extract(file, "^\\d+") %>% as.numeric(),
        tree = file %>% map(parse_ast_4),
        steps = map_int(tree, ~ nrow(.))
    ) %>%
    left_join(read_tsv(here::here("data-raw", "hoc4", "asts", "counts.txt")), by = "astId") %>%
    left_join(read_tsv(here::here("data-raw", "hoc4", "asts", "unitTestResults.txt")), by = "astId") %>%
    arrange(desc(counts), desc(score)) %>%
    select(-file)

traj_4 <- get_traj(here::here("data-raw", "hoc4", "trajectories"))


# problem 18 --------------------------------------------------------------
trees_18 <-
    tibble(
        file = list.files(here::here("data-raw", "hoc18", "asts")) %>% str_subset(".json$")
    ) %>%
    mutate(
        astId = str_extract(file, "^\\d+") %>% as.numeric(),
        tree = file %>% map(parse_ast_18),
        steps = map_int(tree, ~ nrow(.))
    ) %>%
    left_join(read_tsv(here::here("data-raw", "hoc18", "asts", "counts.txt")), by = "astId") %>%
    left_join(read_tsv(here::here("data-raw", "hoc18", "asts", "unitTestResults.txt")), by = "astId") %>%
    arrange(desc(counts), desc(score)) %>%
    select(-file)

traj_18 <- get_traj(here::here("data-raw", "hoc18", "trajectories"))


# save --------------------------------------------------------------------
trees_4 %>% write_rds("data-created/trees_4.rds")
traj_4 %>% write_rds("data-created/traj_4.rds")
traj_18 %>% write_rds("data-created/traj_18.rds")
trees_18 %>% write_rds("data-created/trees_18")

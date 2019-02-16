parse_ast_4 <- function(file){
    print(file)

    ast <-
        file %>%
        here::here("data-raw", "hoc4", "asts", .) %>%
        fromJSON() %>%
        extract2("children") %>%
        as_tibble()

    if("children" %in% names(ast)){

        ast_new <-
            ast %>%
            pull(children) %>%
            extract2(1) %>%
            as_tibble()

        if(!"children" %in% names(ast_new)){
            ast_new$children <- NA
        }

        out <-
            ast_new %>%
            mutate(
                type =
                    map2_chr(
                        type,
                        children,
                        ~ifelse(.x == "maze_turn", .y$type, .x)
                    )
            ) %>%
            select(id, type)

    } else {
        out <- ast %>% as_tibble()
    }

    if(!all(out$type %in% c("maze_moveForward", "turnLeft", "turnRight"))){
        stop("unrecognized type")
    }

    out
}

parse_ast_18 <- function(file){
    print(file)

    file %>%
        here::here("data-raw", "hoc18", "asts", .) %>%
        fromJSON() %>%
        extract2("children") %>%
        as_tibble()
}

get_traj <- function(folder_path){
    tibble(
        trajId = list.files(folder_path) %>% str_subset("^\\d+.txt$") %>% str_extract("^\\d+") %>% as.numeric()
    ) %>%
        mutate(
            astIds = trajId %>% map(~ readLines(paste0(folder_path, "/", ., ".txt"), warn = FALSE))
        )
}

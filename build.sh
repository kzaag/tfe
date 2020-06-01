gcc \
    main.c tfe.c \
    -ltensorflow  \
    -o tfe \
    -Wall -Wextra -Wpedantic -pedantic \
    -std=c99 \
    -Wmissing-prototypes -Wstrict-prototypes -Wold-style-definition \
    `pkg-config --libs opencv`;
    

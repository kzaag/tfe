gcc \
    main.c tfe.c \
    -ltensorflow \
    -o tfe \
    -Wextra -Wpedantic -std=c99 \
    `pkg-config --libs opencv`;

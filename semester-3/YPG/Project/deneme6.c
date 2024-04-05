#include <stdio.h>
#include <stdlib.h>

#define ROW 6
#define COL 10

typedef struct {
    int row;
    int col;
    int distance;
} Node;

typedef struct {
    Node *array;
    int front, rear;
    unsigned capacity;
} Queue;

Queue *createQueue(unsigned capacity) {
    Queue *queue = (Queue *)malloc(sizeof(Queue));
    queue->capacity = capacity;
    queue->front = queue->rear = -1;
    queue->array = (Node *)malloc(capacity * sizeof(Node));
    return queue;
}

int isEmpty(Queue *queue) {
    return queue->front > queue->rear;
}

void enqueue(Queue *queue, int row, int col, int distance) {
    Node newNode;
    newNode.row = row;
    newNode.col = col;
    newNode.distance = distance;

    if (isEmpty(queue)) {
        queue->front = queue->rear = 0;
        queue->array[0] = newNode;
    } else {
        queue->rear++;
        queue->array[queue->rear] = newNode;
    }
}

Node dequeue(Queue *queue) {
    Node emptyNode = {-1, -1, -1};
    if (isEmpty(queue)) {
        return emptyNode;
    }

    Node frontNode = queue->array[queue->front];
    if (queue->front == queue->rear) {
        queue->front = queue->rear = -1;
    } else {
        queue->front++;
    }

    return frontNode;
}

int isValid(int row, int col) {
    return (row >= 0 && row < ROW && col >= 0 && col < COL);
}

int isDestination(int row, int col, int destRow, int destCol) {
    return (row == destRow && col == destCol);
}

void printShortestPath(char grid[ROW][COL], int startRow, int startCol, int destRow, int destCol) {
    Queue *queue = createQueue(ROW * COL);

    int visited[ROW][COL];
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            visited[i][j] = 0;
        }
    }

    enqueue(queue, startRow, startCol, 0);
    visited[startRow][startCol] = 1;

    int rowNum[] = {-1, 0, 0, 1};
    int colNum[] = {0, -1, 1, 0};

    while (!isEmpty(queue)) {
        Node current = dequeue(queue);

        int row = current.row;
        int col = current.col;
        int distance = current.distance;

        if (isDestination(row, col, destRow, destCol)) {
            printf("En kısa yol: %d adım\n", distance);
            return;
        }

        for (int i = 0; i < 4; i++) {
            int newRow = row + rowNum[i];
            int newCol = col + colNum[i];

            if (isValid(newRow, newCol) && grid[newRow][newCol] != '1' && !visited[newRow][newCol]) {
                visited[newRow][newCol] = 1;
                enqueue(queue, newRow, newCol, distance + 1);
            }
        }
    }

    printf("C'ye giden yol bulunamadı.\n");
}

int main() {
    char grid[ROW][COL] = {
        {'0', '0', '1', '0', 'P', '0', 'e', '0', 'p', '0'},
        {'1', '0', 'p', '0', '1', 'E', '1', '0', '1', '0'},
        {'1', 'G', '1', '0', '1', '0', '1', '0', '0', '1'},
        {'0', '0', 'P', '0', 'p', '0', '0', '0', '1', '0'},
        {'0', '1', '1', '0', '1', 'C', '1', '0', '1', '0'},
        {'0', '0', '0', '0', '1', '0', 'p', '0', '0', '0'}
    };

    int startRow = 2; // G'nin satır indeksi
    int startCol = 1; // G'nin sütun indeksi
    int destRow = 4;  // C'nin satır indeksi
    int destCol = 5;  // C'nin sütun indeksi

    printShortestPath(grid, startRow, startCol, destRow, destCol);

    return 0;
}
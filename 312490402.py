from sklearn.datasets import fetch_openml
import numpy as np
import heapq as hq


def knn(train, train_labels, q_image, k):
    knn_heap = [(float('-inf'), -1)] * k  # Using max heap to maintain knn (dist, index)
    hq.heapify(knn_heap)
    labels_counting = [0] * 10  # Counting labels appearances

    for i, image in enumerate(train):
        dist = np.linalg.norm(q_image - image)  # 2-norm
        if dist < -1 * knn_heap[0][0]:  # Insert current index to heap if lower than the max dist in heap
            hq.heappushpop(knn_heap, (-1 * dist, i))
    for item in knn_heap:  # Count labels appearances from the knn heap
        digit = int(train_labels[item[1]])
        labels_counting[digit] += 1
    max_app = max(labels_counting)
    # Array of the all the majority labels (could be only one)
    max_digits = [digit for digit, occ in enumerate(labels_counting) if occ == max_app]
    return np.random.choice(max_digits)  # Choose arbitrarily the label among the majority ones


def acc_knn(n, k, train, train_labels, test, test_labels):
    acc = 0
    for i, image in enumerate(test):
        if int(test_labels[i]) == knn(train[:n], train_labels[:n], image, k):
            acc += 1
    return acc / len(test)


def main():
    mnist = fetch_openml("mnist_784", as_frame=False)
    data = mnist['data']
    labels = mnist['target']
    idx = np.random.RandomState(0).choice(70000, 11000)
    train = data[idx[:10000], :].astype(int)
    train_labels = labels[idx[:10000]]
    test = data[idx[10000:], :].astype(int)
    test_labels = labels[idx[10000:]]

    print(acc_knn(1000, 10, train, train_labels, test, test_labels))


if __name__ == "__main__":
    main()

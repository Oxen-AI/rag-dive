
def generate_batches(dataset, batch_size):
    batch = []
    for x in dataset:
        batch.append(x)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

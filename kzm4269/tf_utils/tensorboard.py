from tensorboard.backend.event_processing import event_accumulator


def scalars(path, tag):
    ea = event_accumulator.EventAccumulator(path, size_guidance={
        event_accumulator.COMPRESSED_HISTOGRAMS: 1,
        event_accumulator.HISTOGRAMS: 1,
        event_accumulator.IMAGES: 1,
        event_accumulator.AUDIO: 1,
        event_accumulator.SCALARS: 0,
        event_accumulator.TENSORS: 1,
        event_accumulator.GRAPH: 1,
        event_accumulator.META_GRAPH: 1,
        event_accumulator.RUN_METADATA: 1,
    })
    ea.Reload()

    return ea.Scalars(tag)

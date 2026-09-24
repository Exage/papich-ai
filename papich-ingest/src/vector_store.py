from uuid import NAMESPACE_URL, uuid5

COLLECTION = 'video_chunks'


class VectorStore:
    def __init__(self, path, dimension=None):
        from qdrant_client import QdrantClient, models
        self.models = models
        self.client = QdrantClient(path=str(path))
        if dimension and not self.client.collection_exists(COLLECTION):
            self.client.create_collection(COLLECTION, vectors_config=models.VectorParams(
                size=dimension, distance=models.Distance.COSINE))
        if dimension:
            actual = self.client.get_collection(COLLECTION).config.params.vectors.size
            if actual != dimension:
                self.close()
                raise ValueError('Vector dimension mismatch; use a new data directory')

    def exists(self):
        return self.client.collection_exists(COLLECTION)

    def video_filter(self, video_id):
        return self.models.Filter(must=[self.models.FieldCondition(
            key='video_id', match=self.models.MatchValue(value=video_id))])

    def count_video(self, video_id):
        return self.client.count(COLLECTION, count_filter=self.video_filter(video_id), exact=True).count if self.exists() else 0

    def delete_video(self, video_id):
        if self.exists():
            self.client.delete(COLLECTION, points_selector=self.models.FilterSelector(
                filter=self.video_filter(video_id)), wait=True)

    def upsert(self, chunks, vectors):
        if len(chunks) != len(vectors):
            raise ValueError('Chunk/vector length mismatch')
        points = [self.models.PointStruct(id=str(uuid5(NAMESPACE_URL, c['id'])),
                                          vector=v, payload=c)
                  for c, v in zip(chunks, vectors)]
        if points:
            self.client.upsert(COLLECTION, points=points, wait=True)

    def search(self, vector, top_k, video_ids=None):
        if not self.exists() or video_ids == []:
            return []
        query_filter = None
        if video_ids is not None:
            query_filter = self.models.Filter(must=[self.models.FieldCondition(
                key='video_id', match=self.models.MatchAny(any=video_ids))])
        return self.client.query_points(COLLECTION, query=vector, limit=top_k,
                                        query_filter=query_filter, with_payload=True).points

    def close(self):
        self.client.close()

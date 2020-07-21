class MetaPredicateData(object):
    @staticmethod
    def get_predicate_names(predicate_lib):
        return list(predicate_lib.relation_names)

    @staticmethod
    def get_predicates(predicate_lib):
        predicates = [getattr(predicate_lib, attr)
                      for attr in predicate_lib.relation_names]
        return predicates

    @staticmethod
    def get_predicate_count(predicate_lib):
        return len(predicate_lib.relation_names)

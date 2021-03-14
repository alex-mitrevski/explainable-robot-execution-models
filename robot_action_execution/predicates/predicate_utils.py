class PredicateLibraryBase(object):
    relation_names = None

    relation_parameter_causes = None

    disjoint_predicates = None


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

    @staticmethod
    def get_predicate_parameter_cause(predicate_lib, predicate):
        assert predicate in MetaPredicateData.get_predicate_names(predicate_lib), \
               '{0} does not exist in {1}'.format(predicate, predicate_lib)

        assert predicate_lib.relation_parameter_causes is not None, \
               'Relation causes are not specified for {0}'.format(predicate_lib)

        assert isinstance(predicate_lib.relation_parameter_causes, dict), \
               'relation_parameter_causes is not of type dict for {0}'.format(predicate_lib)

        parameter_cause = None
        for (parameter, relations) in predicate_lib.relation_parameter_causes.items():
            if predicate in relations:
                parameter_cause = parameter
                break

        if parameter_cause is None:
            raise RuntimeError('The parameter that determines {0} is not specified'.format(predicate))
        return parameter_cause

    @staticmethod
    def get_disjoint_predicates(predicate_lib, predicate):
        assert predicate in MetaPredicateData.get_predicate_names(predicate_lib), \
               '{0} does not exist in {1}'.format(predicate, predicate_lib)

        assert predicate_lib.disjoint_predicates is not None, \
               'The disjoint predicates are not specified for {0}'.format(predicate_lib)

        assert isinstance(predicate_lib.disjoint_predicates, list), \
               'relation_parameter_causes is not of type list for {0}'.format(predicate_lib)

        disjoint_predicates = []
        for predicate_list in predicate_lib.disjoint_predicates:
            if predicate in predicate_list:
                disjoint_predicates.append(predicate_list)
        return disjoint_predicates

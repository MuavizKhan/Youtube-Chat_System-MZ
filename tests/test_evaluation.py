import pytest
from Backend.rag_system.evaluation import (
    EXPECTED_RETRIEVAL,TEST_QUESTIONS,build_evaluation_summary,evaluate_retrieval_result
)


@pytest.mark.regression
def test_ids_and_expectations_are_complete():
    ids=[item["id"] for item in TEST_QUESTIONS]
    assert len(ids)==len(set(ids))==13
    assert set(EXPECTED_RETRIEVAL)==set(ids)
    assert EXPECTED_RETRIEVAL["Q01"]=="must_retrieve"
    assert EXPECTED_RETRIEVAL["Q11"]=="must_retrieve"
    assert EXPECTED_RETRIEVAL["Q12"]=="may_retrieve"
    assert EXPECTED_RETRIEVAL["Q13"]=="must_not_retrieve"


@pytest.mark.regression
@pytest.mark.parametrize("question_id,count,expected",[
    ("Q01",1,True),("Q01",0,False),("Q12",0,True),("Q12",4,True),
    ("Q13",0,True),("Q13",1,False),
])
def test_expectation_contract(question_id,count,expected):
    passed,_=evaluate_retrieval_result(question_id,count)
    assert passed is expected


@pytest.mark.regression
def test_negative_retrieval_count_is_rejected():
    with pytest.raises(ValueError):
        evaluate_retrieval_result("Q01",-1)


@pytest.mark.regression
def test_summary_metrics():
    results=[
        {"id":"Q01","expected":"must_retrieve","passed":True,"retrieved":2},
        {"id":"Q12","expected":"may_retrieve","passed":True,"retrieved":0},
        {"id":"Q13","expected":"must_not_retrieve","passed":True,"retrieved":0},
    ]
    summary=build_evaluation_summary("MdeQMVBuGgY",results)
    assert summary["total"]==3
    assert summary["passed"]==3
    assert summary["failed"]==0
    assert summary["pass_rate"]==pytest.approx(1.0)
    assert summary["strict_total"]==2
    assert summary["strict_passed"]==2
    assert summary["strict_pass_rate"]==pytest.approx(1.0)

namespace HRMS.Models.Responses;

public class GetAvailableSettingForMemberOperationResponse
{
    public List<DepartmentResponse> Departments { get; set; }
    
    public List<TeamInfo> Teams { get; set; }
    
    public List<PositionResponse> Positions { get; set; }
    
    public List<JobGradeResponse> JobGrades { get; set; }
}
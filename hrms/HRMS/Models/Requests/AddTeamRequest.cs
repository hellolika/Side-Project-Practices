using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class AddTeamRequest
{
    public string TeamName { get; set; }
    public int DepartmentId { get; set; }
    public TimeSpan StartTime { get; set; }
    public TimeSpan EndTime { get; set; }
    public decimal TotalHour { get; set; }
    
    [JsonIgnore]
    public int CreatedBy { get; set; }
}
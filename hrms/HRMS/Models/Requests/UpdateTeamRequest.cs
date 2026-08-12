using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class UpdateTeamRequest
{
    public int TeamId { get; set; }
    public string TeamName { get; set; }
    public int DepartmentId { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public decimal TotalHour { get; set; }
    
    [JsonIgnore]
    public int ModifiedBy { get; set; }
}
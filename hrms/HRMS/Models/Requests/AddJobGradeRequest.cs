using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class AddJobGradeRequest
{
    public string JobGradeName { get; set; }
    public int PositionId { get; set; }
    
    [JsonIgnore]
    public int CreatedBy { get; set; }
}
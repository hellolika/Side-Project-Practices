using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class UpdateJobGradeRequest
{
    public int JobGradeId { get; set; }
    public string JobGradeName { get; set; }
    public int PositionId { get; set; }
    
    [JsonIgnore]
    public int ModifiedBy { get; set; }
}
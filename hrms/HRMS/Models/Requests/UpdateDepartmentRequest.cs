using Newtonsoft.Json;

namespace HRMS.Models.Requests;

public class UpdateDepartmentRequest
{
    public int DepartmentId { get; set; }
    public string DepartmentName { get; set; }
    
    [JsonIgnore]
    public int ModifiedBy { get; set; }
}
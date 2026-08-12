CREATE PROCEDURE [dbo].[HRMS_GetAnnouncements.2.0.0]
	@page AS INT,
	@itemPerPage AS INT
AS
BEGIN 
	SET NOCOUNT ON

	SELECT a.[Id], a.[Title], a.[Message], a.[CreatedOn],
	c.[Username] AS [CreatedBy], a.[ModifiedOn], m.[Username] AS [ModifiedBy]
	FROM [dbo].[Announcement] a WITH(NOLOCK)
	LEFT JOIN [dbo].[Member] c ON a.[CreatedBy] = c.[Id]
	LEFT JOIN [dbo].[Member] m ON a.[ModifiedBy] = m.[Id]
	ORDER BY a.[CreatedOn] DESC
	OFFSET (@page-1)*@itemPerPage ROWS
	FETCH NEXT @itemPerPage ROWS ONLY
END